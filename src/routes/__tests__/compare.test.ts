import { Request, Response } from 'express';
import { compareAddresses } from '../compare';
import { findAddressMatches } from '../../services/database.service';
import { AddressMatch } from '../../types/address';
import { createMockRequest, createMockResponse } from '../../test/helpers';

jest.mock('../../services/database.service');

describe('Compare Addresses Endpoint', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let responseObject: { [key: string]: any };

  beforeEach(() => {
    responseObject = {};
    mockRequest = createMockRequest({
      sourceFile: [
        { address: '123 Main St', city: 'Test City', state: 'TS', zip: '12345' }
      ],
      columns: ['address', 'city', 'state', 'zip'],
      threshold: 80
    });

    mockResponse = createMockResponse(responseObject);

    (findAddressMatches as jest.Mock).mockReset();
  });

  test('should return matches when found', async () => {
    const mockMatches = [{
      raw_address: '123 Main Street',
      member_id: 'TEST123',
      lic: 'LIC456',
      match_score: 0.85
    }];

    (findAddressMatches as jest.Mock).mockResolvedValueOnce(mockMatches);

    await compareAddresses(mockRequest as Request, mockResponse as Response);

    expect(responseObject).toEqual({
      status: 'success',
      data: [{
        original_row: mockRequest.body.sourceFile[0],
        matches: mockMatches
      }]
    });
  });

  test('should handle no matches found', async () => {
    (findAddressMatches as jest.Mock).mockResolvedValueOnce([]);

    await compareAddresses(mockRequest as Request, mockResponse as Response);

    expect(responseObject).toEqual({
      status: 'success',
      data: []
    });
  });

  test('should handle multiple matches for single address', async () => {
    const mockMatches = [
      {
        raw_address: '123 Main Street',
        member_id: 'TEST123',
        lic: 'LIC456',
        match_score: 0.95
      },
      {
        raw_address: '123 Main St',
        member_id: 'TEST124',
        lic: 'LIC457',
        match_score: 0.85
      }
    ];

    (findAddressMatches as jest.Mock).mockResolvedValueOnce(mockMatches);

    await compareAddresses(mockRequest as Request, mockResponse as Response);

    expect(responseObject).toEqual({
      status: 'success',
      data: [{
        original_row: mockRequest.body.sourceFile[0],
        matches: mockMatches
      }]
    });
  });

  test('should handle threshold filtering', async () => {
    mockRequest.body.threshold = 90; // Higher threshold
    const mockMatches: AddressMatch[] = [];

    (findAddressMatches as jest.Mock).mockResolvedValueOnce(mockMatches);

    await compareAddresses(mockRequest as Request, mockResponse as Response);

    expect(responseObject).toEqual({
      status: 'success',
      data: []
    });
  });
});