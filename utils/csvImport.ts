import { parse } from 'csv-parse';
import { createReadStream } from 'fs';
import { pool } from '../config/database.config';
import { pipeline } from 'stream/promises';

interface RosterRecord {
    'Member ID': string;
    'Member Name': string;
    'Address1': string;
    'Address2': string;
    'City': string;
    'State': string;
    'Zip Code': string;
    [key: string]: string;  // Allow other fields
}

export async function importCsvToDatabase(filePath: string): Promise<void> {
    console.log('Starting CSV import from:', filePath);
    
    const parser = parse({
        columns: true,
        skip_empty_lines: true,
        trim: true
    });

    let recordCount = 0;

    try {
        const client = await pool.connect();
        console.log('Database connection successful');

        // First, log the CSV structure
        const firstChunk = await new Promise<string[]>((resolve) => {
            const readStream = createReadStream(filePath);
            let headers = '';
            readStream.on('data', (chunk) => {
                headers += chunk.toString();
                readStream.destroy();
                resolve(headers.split('\n')[0].split(','));
            });
        });
        console.log('CSV Headers:', firstChunk);

        await pipeline(
            createReadStream(filePath),
            parser,
            async function*(source) {
                for await (const record of source) {
                    // Log first record to see structure
                    if (recordCount === 0) {
                        console.log('Sample record:', record);
                    }

                    // Combine address components
                    const address1 = record['Address1'] || '';
                    const address2 = record['Address2'] || '';
                    const fullAddress = [address1, address2].filter(Boolean).join(' ');
                    
                    const components = {
                        street_address: fullAddress,
                        city: record['City'] || '',
                        state: record['State'] || '',
                        postal_code: record['Zip Code'] || ''
                    };

                    const query = `
                        INSERT INTO addresses (
                            raw_address,
                            normalized_address,
                            components,
                            metadata
                        ) VALUES ($1, $2, $3, $4)
                    `;

                    const formattedAddress = `${fullAddress}, ${components.city}, ${components.state} ${components.postal_code}`.trim();
                    
                    const values = [
                        formattedAddress,
                        formattedAddress, // Will normalize later
                        JSON.stringify(components),
                        JSON.stringify({ 
                            source: 'roster_import',
                            imported_at: new Date(),
                            member_id: record['Member ID'],
                            member_name: record['Member Name'],
                            original_record: record
                        })
                    ];

                    if (formattedAddress) {
                        await client.query(query, values);
                        recordCount++;
                        
                        if (recordCount % 100 === 0) {
                            console.log(`Imported ${recordCount} records...`);
                        }
                    }
                    
                    yield record;
                }
            }
        );

        console.log(`CSV import completed. Total records imported: ${recordCount}`);
        client.release();
    } catch (error) {
        console.error('Error importing CSV:', error);
        throw error;
    } finally {
        await pool.end();
    }
}