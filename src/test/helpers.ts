import { Request, Response } from 'express';

export function createMockRequest(data: any): Partial<Request> {
    return {
        body: data
    };
}

export function createMockResponse(responseObject: {[key: string]: any} = {}): Partial<Response> {
    return {
        json: jest.fn().mockImplementation(result => {
            Object.assign(responseObject, result);
            return responseObject;
        }),
        status: jest.fn().mockReturnThis()
    };
}